import os
import re
import time
import json
import base64
from PIL import Image
from io import BytesIO
from collections import defaultdict
import pandas as pd
import arxiv as arxiv


from nv_ingest_client.client import NvIngestClient
from nv_ingest_client.message_clients.rest.rest_client import RestClient
from nv_ingest_client.primitives import JobSpec
from nv_ingest_client.primitives.tasks import DedupTask, ExtractTask, FilterTask
from nv_ingest_client.util.file_processing.extract import extract_file_content


# Constants for configuration and paths
HTTP_HOST = os.environ.get('HTTP_HOST', "localhost")
HTTP_PORT = os.environ.get('HTTP_PORT', "7670")
TASK_QUEUE = os.environ.get('TASK_QUEUE', "morpheus_task_queue")

MINIO_ACCESS_KEY = os.environ.get('MINIO_ACCESS_KEY', "minioadmin")
MINIO_SECRET_KEY = os.environ.get('MINIO_SECRET_KEY', "minioadmin")

DEFAULT_JOB_TIMEOUT = 10000  # Timeout for job completion (in ms)

SCRIPT_DIR_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR_PATH, "sources")


# --- Helper Functions --- #

def parse_id(input_string):
    """
    Parse arXiv ID from either a direct ID string or an arXiv URL.
    """
    # Pattern to match a direct arXiv ID
    id_pattern = re.compile(r"\d{4}\.\d{4,5}(v\d+)?$")
    if id_pattern.match(input_string):
        return input_string

    # Pattern to match an arXiv URL and extract the ID
    url_pattern = re.compile(
        r"https?://(?:www\.)?arxiv\.org/(abs|pdf)/(\d{4}\.\d{4,5})(v\d+)?(\.pdf)?$"
    )
    url_match = url_pattern.match(input_string)
    if url_match:
        return url_match.group(2) + (url_match.group(3) if url_match.group(3) else "")

    # Raise an error if the input does not match any of the expected formats
    raise ValueError(
        f"The provided input '{input_string}' does not match the expected URL or ID format."
    )


def download_arxiv_data():
    """
    Download arXiv articles from URLs listed in arxiv_urls.jsonl and save them as PDFs.
    """
    pdf_root_dir = os.path.join(DATA_DIR, "pdfs")
    os.makedirs(pdf_root_dir, exist_ok=True)

    source_links_file = os.path.join(DATA_DIR, 'arxiv_urls.jsonl')
    if not os.path.exists(source_links_file):
        raise FileNotFoundError(f"File '{source_links_file}' not found.")
    
    urls = pd.read_json(path_or_buf=source_links_file, lines=True)
    urls = urls[0].tolist()

    for url in urls:
        pdf_name = os.path.basename(url)
        pdf_file = os.path.join(pdf_root_dir, pdf_name)

        if os.path.exists(pdf_file):
            print(f"Article '{url}' already exists, skipping download.")
        else:
            article_id = parse_id(url)
            search_result = arxiv.Client().results(arxiv.Search(id_list=[article_id]))

            if article := next(search_result):
                print(f'Downloading arXiv article "{url}"...')
                article.download_pdf(dirpath=pdf_root_dir, filename=pdf_name)
            else:
                print(f"Failed to download article '{url}'.")
                return None


def extract_contents():
    """
    Extract contents from downloaded PDFs and send them for processing.
    """
    downloaded_path = os.path.join(DATA_DIR, "pdfs")
    result_root_dir = os.path.join(DATA_DIR, "extracted_data")
    os.makedirs(result_root_dir, exist_ok=True)

    for file in os.listdir(downloaded_path):
        sample_pdf = os.path.join(downloaded_path, file)
        file_content, file_type = extract_file_content(sample_pdf)
        
        # Initialize the client
        client = NvIngestClient(
            message_client_allocator=RestClient,
            message_client_hostname=HTTP_HOST,
            message_client_port=HTTP_PORT,
            message_client_kwargs=None,
            msg_counter_id="nv-ingest-message-id",
            worker_pool_size=1,
        )

        job_spec = JobSpec(
            document_type=file_type,
            payload=file_content,
            source_id=sample_pdf,
            source_name=sample_pdf,
            extended_options={
                "tracing_options": {
                    "trace": True,
                    "ts_send": time.time_ns(),
                }
            },
        )

        # Define tasks
        extract_task = ExtractTask(
            document_type=file_type,
            extract_text=True,
            extract_images=True,
            extract_tables=True,
            extract_charts=True,
            text_depth="document",
            extract_tables_method="yolox",
        )

        dedup_task = DedupTask(
            content_type="image",
            filter=True,
        )

        # filter_task = FilterTask(
        #     content_type="image",
        #     min_size=28,
        #     max_aspect_ratio=5.0,
        #     min_aspect_ratio=0.2,
        #     filter=True,
        # )

        job_spec.add_task(extract_task)
        job_spec.add_task(dedup_task)
        # job_spec.add_task(filter_task)

        # Submit the job and retrieve results
        job_id = client.add_job(job_spec)
        client.submit_job(job_id, TASK_QUEUE)

        generated_metadata = client.fetch_job_result(job_id, timeout=DEFAULT_JOB_TIMEOUT)[0][0]

        # Save extracted metadata
        with open(os.path.join(result_root_dir, f'generated_metadata_{file}.json'), 'w') as f:
            json.dump(generated_metadata, f, indent=4)


def analyze_contents():
    """
    Analyze the extracted contents for unique types and descriptions.
    """
    result_root_dir = os.path.join(DATA_DIR, "extracted_data")
    unique_types = set()
    unique_desc = set()
    type_indices = defaultdict(list)

    # Iterate through the generated metadata files
    for file in os.listdir(result_root_dir):
        file_path = os.path.join(result_root_dir, file)
        with open(file_path, 'r') as f:
            generated_metadata = json.load(f)

        for i in range(len(generated_metadata)):
            description = generated_metadata[i]['metadata']['content_metadata']['description']
            unique_desc.add(description)

            content_type = generated_metadata[i]['metadata']['content_metadata']['type']
            unique_types.add(content_type)
            type_indices[content_type].append(i)

    print("Unique types:", unique_types)
    for content_type, indices in type_indices.items():
        print(f"{content_type}: {len(indices)}")
    

def display_contents():
    """
    Display the extracted image or table contents as base64 decoded files.
    """
    result_root_dir = os.path.join(DATA_DIR, "extracted_data")

    output_folder = os.path.join(DATA_DIR, "extracted_display_data")
    os.makedirs(output_folder, exist_ok=True)

    for file in os.listdir(result_root_dir):
        file_path = os.path.join(result_root_dir, file)
        with open(file_path, 'r') as f:
            generated_metadata = json.load(f)
        
        for idx, metadata in enumerate(generated_metadata):
            content = metadata["metadata"]['content_metadata']['type']
            if content in ['image', 'table']:
                # Decode and save image data
                image_data_b64 = metadata["metadata"]["content"]
                image_data = base64.b64decode(image_data_b64)
                image = Image.open(BytesIO(image_data))
                image_filename = os.path.join(output_folder, f'{file}_{idx}.png')
                image.save(image_filename, format='PNG')

# --- Main Execution --- #

def main():
    """
    Main function to execute the workflow.
    """
    download_arxiv_data()
    extract_contents()
    analyze_contents()
    display_contents()


if __name__ == "__main__":
    main()
