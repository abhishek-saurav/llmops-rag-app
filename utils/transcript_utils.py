from pathlib import Path 
import re 

TIMESTAMP_PATTERN = re.compile(r"^\d{2}:\d{2}:\d{2}\.\d{3}\s-->\s+\d{2}:\d{2}:\d{2}\.\d{3}$")

def clean_transcript_text(transcript: str) -> str: 
    lines = transcript.splitlines()
    cleaned_lines = [] 

    for line in lines: 
        stripped = line.strip() 
        if not stripped: 
            continue 

        if TIMESTAMP_PATTERN.match(stripped): 
            continue 

        cleaned_lines.append(stripped)

    return "\n".join(cleaned_lines)


def load_and_clean_transcript(original_dir: Path | str, new_dir: Path | str): 

    ORIGINAL_PATH = Path(original_dir) 
    NEW_PATH = Path(new_dir)

    # create the new directory 

    NEW_PATH.mkdir(parents= True, exist_ok= True)

    transcripts = ORIGINAL_PATH.glob(pattern= "*.vtt")
    for transcript_path in transcripts: 
        # read the text file 
        file_content = transcript_path.read_text(encoding= 'utf-8')

        # clean the text 
        cleaned_content = clean_transcript_text(file_content)

        # get the file name 
        # file_name = transcript_path.name 
        file_name = transcript_path.with_suffix(".txt").name


        # create the output path 
        output_path = NEW_PATH / file_name

        # write the file contne t
        output_path.write_text(cleaned_content, encoding= 'utf-8')

RAW_PATH = Path(__file__).parent.parent / "data" / "raw"
PROCESSED_PATH  = RAW_PATH.parent / "processed"
if __name__ == "__main__": 
    load_and_clean_transcript(RAW_PATH, PROCESSED_PATH)