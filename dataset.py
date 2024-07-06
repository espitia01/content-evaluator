from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import json
from datetime import datetime, timedelta, timezone
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Replace with your actual API key
API_KEY = 'AIzaSyALp7_LGzCgD1Tf9Uv2YGuJ2TzI_MeWRm8'

# Create a YouTube API client
youtube = build('youtube', 'v3', developerKey=API_KEY)

def get_sports_videos_data(region_code, max_results=50, page_token=None):
    videos_data = []
    
    # Calculate the start and end of the day 7 days ago
    seven_days_ago = datetime.now(timezone.utc) - timedelta(days=7)
    start_of_day = seven_days_ago.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
    end_of_day = seven_days_ago.replace(hour=23, minute=59, second=59, microsecond=999999).isoformat()

    try:
        logging.info(f"Fetching sports videos for region: {region_code}")
        # Search for YouTube sports videos
        search_response = youtube.search().list(
            type='video',
            part='id,snippet',
            maxResults=max_results,
            regionCode=region_code,
            publishedAfter=start_of_day,
            publishedBefore=end_of_day,
            order='date',
            videoCategoryId='17',  # Category ID for Sports
            pageToken=page_token
        ).execute()

        logging.info(f"Found {len(search_response['items'])} videos in search results")

        for item in search_response['items']:
            video_id = item['id']['videoId']
            
            try:
                logging.info(f"Fetching details for video ID: {video_id}")
                # Get video statistics and details
                video_response = youtube.videos().list(
                    part='statistics,contentDetails,snippet',
                    id=video_id
                ).execute()

                if not video_response['items']:
                    logging.warning(f"No details found for video ID: {video_id}")
                    continue

                video_stats = video_response['items'][0].get('statistics', {})
                content_details = video_response['items'][0].get('contentDetails', {})
                video_snippet = video_response['items'][0].get('snippet', {})

                # Get channel details
                channel_id = item['snippet']['channelId']
                logging.info(f"Fetching details for channel ID: {channel_id}")
                channel_response = youtube.channels().list(
                    part='statistics,snippet',
                    id=channel_id
                ).execute()

                if not channel_response['items']:
                    logging.warning(f"No details found for channel ID: {channel_id}")
                    continue

                channel_stats = channel_response['items'][0].get('statistics', {})
                channel_snippet = channel_response['items'][0].get('snippet', {})

                videos_data.append({
                    'video_id': video_id,
                    'video_url': f'https://www.youtube.com/watch?v={video_id}',
                    'title': item['snippet']['title'],
                    'description': item['snippet']['description'],
                    'channel_name': item['snippet']['channelTitle'],
                    'published_at': item['snippet']['publishedAt'],
                    'view_count': int(video_stats.get('viewCount', 0)),
                    'like_count': int(video_stats.get('likeCount', 0)),
                    'comment_count': int(video_stats.get('commentCount', 0)),
                    'duration': content_details.get('duration', 'N/A'),
                    'channel_subscriber_count': int(channel_stats.get('subscriberCount', 0)),
                    'channel_video_count': int(channel_stats.get('videoCount', 0)),
                    'channel_view_count': int(channel_stats.get('viewCount', 0)),
                    'channel_created_at': channel_snippet.get('publishedAt', 'N/A'),
                    'tags': video_snippet.get('tags', []),
                    'region': region_code,
                    'collection_timestamp': datetime.now().isoformat() + 'Z'
                })
                logging.info(f"Successfully collected data for video ID: {video_id}")
            except HttpError as e:
                logging.error(f"An error occurred while fetching details for video ID {video_id}: {e}")
                continue

        logging.info(f"Collected data for {len(videos_data)} videos from region {region_code}")
        return videos_data, search_response.get('nextPageToken')

    except HttpError as e:
        if e.resp.status == 403 and "quotaExceeded" in str(e):
            logging.warning(f"Quota exceeded for region {region_code}")
            raise  # Re-raise the error
        logging.error(f'An HTTP error {e.resp.status} occurred: {e.content}')
        return [], None

def collect_large_dataset(target_size=10000):
    regions = ['AR']  # Argentina
    combined_dataset = []
    page_tokens = {region: None for region in regions}
    processed_video_ids = set()  # Set to store processed video IDs

    logging.info(f"Starting to collect a dataset of {target_size} videos")
    try:
        while len(combined_dataset) < target_size:
            for region in regions:
                logging.info(f"Collecting data for region: {region}")
                videos, next_page_token = get_sports_videos_data(region, max_results=50, page_token=page_tokens[region])
                
                # Filter out duplicates
                new_videos = [video for video in videos if video['video_id'] not in processed_video_ids]
                
                combined_dataset.extend(new_videos)
                processed_video_ids.update(video['video_id'] for video in new_videos)
                
                page_tokens[region] = next_page_token

                logging.info(f"Total unique videos collected so far: {len(combined_dataset)}")

                if len(combined_dataset) >= target_size:
                    logging.info("Reached target dataset size")
                    break

                # Sleep to avoid hitting API rate limits
                logging.info("Sleeping for 1 second to avoid API rate limits")
                time.sleep(1)

    except HttpError as e:
        if e.resp.status == 403 and "quotaExceeded" in str(e):
            logging.warning("API quota exceeded. Saving current data and exiting.")
        else:
            logging.error(f"An unexpected error occurred: {e}")
    finally:
        # Always save the data, whether we've completed or encountered an error
        with open('youtube_sports_videos.json', 'w', encoding='utf-8') as f:
            json.dump(combined_dataset, f, indent=2, ensure_ascii=False)
        logging.info(f'Dataset saved with {len(combined_dataset)} unique YouTube sports videos')

    return combined_dataset[:target_size]

if __name__ == "__main__":
    logging.info("Starting data collection process")
    collected_dataset = collect_large_dataset(3000)
    logging.info(f"Data collection completed with {len(collected_dataset)} videos")