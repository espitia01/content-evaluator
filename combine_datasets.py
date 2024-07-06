import json
import os

def combine_json_datasets(file_paths):
    combined_data = []
    seen_video_ids = set()

    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            for item in data:
                if item['video_id'] not in seen_video_ids:
                    combined_data.append(item)
                    seen_video_ids.add(item['video_id'])

    return combined_data

def main():
    file_paths = [
        '/Users/giovannyespitia/Documents/projects/tiktok-predictor-v0/youtube_sports_videos.json',
        '/Users/giovannyespitia/Documents/projects/tiktok-predictor-v0/combined_youtube_sports_videos_dataset.json'
    ]

    combined_data = combine_json_datasets(file_paths)

    output_path = '/Users/giovannyespitia/Documents/projects/tiktok-predictor-v0/combined_youtube_sports_videos_dataset_2.json'
    with open(output_path, 'w', encoding='utf-8') as outfile:
        json.dump(combined_data, outfile, indent=2, ensure_ascii=False)

    print(f"Combined dataset saved to: {output_path}")
    print(f"Total number of unique videos in the combined dataset: {len(combined_data)}")

if __name__ == "__main__":
    main()