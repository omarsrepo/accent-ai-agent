"""
Video processing utilities for accent detection system.
Handles video URL input and audio extraction with proxy service support.
"""

import os
import tempfile
import logging
from pathlib import Path
import yt_dlp
import ffmpeg
import requests
import json
import re
import urllib.parse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VideoProcessor:
    """Class to handle video URL input and audio extraction using proxy services."""
    
    def __init__(self):
        """
        Initialize the VideoProcessor.
        Creates a temporary directory to save extracted audio files to and logs it.
        """
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.temp_dir.name)
        logger.info(f"Output directory set to: {self.output_dir}")
    
    def __del__(self):
        """Clean up temporary directory if it was created."""
        if hasattr(self, 'temp_dir'):
            self.temp_dir.cleanup()
    
    def extract_audio_from_url(self, video_url):
        """
        Extract audio from a video URL.
        
        Args:
            video_url (str): URL of the video to process.
            
        Returns:
            str: Path to the extracted audio file.
            
        Raises:
            ValueError: If the URL is invalid or unsupported.
            RuntimeError: If audio extraction fails.
        """
        logger.info(f"Processing video URL: {video_url}")
        
        # Validate URL
        if not video_url or not isinstance(video_url, str):
            raise ValueError("Invalid URL provided")
        
        try:
            # First, download the video or get direct video file URL
            video_path = self._download_or_get_video(video_url)
            
            # Then extract audio from the video
            audio_path = self._extract_audio(video_path)
            
            return audio_path
            
        except Exception as e:
            logger.error(f"Error processing video URL: {str(e)}")
            raise RuntimeError(f"Failed to process video URL: {str(e)}")
    
    def _download_or_get_video(self, url):
        """
        Download video from URL or get direct file path using proxy services when needed.
        
        Args:
            url (str): Video URL.
            
        Returns:
            str: Path to the downloaded or direct video file.
        """
        logger.info(f"Downloading or accessing video from: {url}")
        
        # Check if it's a direct file URL (ends with common video extensions)
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
        if any(url.lower().endswith(ext) for ext in video_extensions):
            # For direct video URLs, download the file
            output_path = self.output_dir / f"video{Path(url).suffix}"
            
            ydl_opts = {
                'format': 'bestaudio[ext=m4a]/best[ext=mp4]/best',
                'outtmpl': str(output_path),
                'quiet': False,
                'noplaylist': True,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            
            return str(output_path)
        
        # For YouTube URLs, try multiple approaches
        output_path = self.output_dir / "video.mp4"
        
        # First attempt: Try standard yt-dlp download
        try:
            logger.info("Attempting standard download using yt-dlp")
            ydl_opts = {
                'format': 'bestaudio[ext=m4a]/best[ext=mp4]/best',
                'outtmpl': str(output_path),
                'quiet': False,
                'noplaylist': True,
                'ignoreerrors': True,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
                
            # Check if the file was downloaded successfully
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                logger.info("Standard download successful")
                return str(output_path)
        except Exception as e:
            logger.warning(f"Standard download failed: {str(e)}")
        
        # Second attempt: Try using a proxy service
        try:
            logger.info("Attempting download via proxy service")
            
            # Extract video ID from YouTube URL
            video_id = self._extract_youtube_id(url)
            if not video_id:
                raise ValueError("Could not extract YouTube video ID")
                
            # Try multiple proxy services
            proxy_methods = [
                self._download_via_invidious,
                self._download_via_y2mate,
                self._download_via_alternative_domain
            ]
            
            for method in proxy_methods:
                try:
                    result = method(video_id, output_path)
                    if result and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                        logger.info(f"Proxy download successful using {method.__name__}")
                        return str(output_path)
                except Exception as proxy_e:
                    logger.warning(f"{method.__name__} failed: {str(proxy_e)}")
                    continue
                    
            # If all proxy methods failed, try direct audio URL
            audio_path = self.output_dir / "audio.wav"
            if self._download_direct_audio_url(video_id, audio_path):
                return str(audio_path)
                
        except Exception as e:
            logger.warning(f"Proxy service download failed: {str(e)}")
        
        # Third attempt: Try with simpler options as a last resort
        logger.info("Trying fallback download options")
        fallback_ydl_opts = {
            'format': 'worstaudio/worst',  # Try worst quality as a last resort
            'outtmpl': str(output_path),
            'quiet': False,
            'noplaylist': True,
        }
        
        with yt_dlp.YoutubeDL(fallback_ydl_opts) as ydl:
            ydl.download([url])
        
        # Check if any file was downloaded
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            # Try one more time with a different URL format (sometimes helps with YouTube)
            if 'youtu.be' in url:
                new_url = url.replace('youtu.be/', 'youtube.com/watch?v=')
                logger.info(f"Trying alternative URL format: {new_url}")
                with yt_dlp.YoutubeDL(fallback_ydl_opts) as ydl:
                    ydl.download([new_url])
            
            # If still no file, raise error
            if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
                raise RuntimeError("Failed to download video after multiple attempts. Try a different video or manually download the audio.")
            
        return str(output_path)
    
    def _extract_youtube_id(self, url):
        """
        Extract YouTube video ID from URL.
        
        Args:
            url (str): YouTube URL.
            
        Returns:
            str: YouTube video ID or None if not found.
        """
        # Regular expressions for different YouTube URL formats
        patterns = [
            r'(?:youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]{11})',
            r'youtube\.com/embed/([a-zA-Z0-9_-]{11})',
            r'youtube\.com/v/([a-zA-Z0-9_-]{11})',
            r'youtube\.com/shorts/([a-zA-Z0-9_-]{11})'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
                
        return None
    
    def _download_via_invidious(self, video_id, output_path):
        """
        Download video using Invidious proxy.
        
        Args:
            video_id (str): YouTube video ID.
            output_path (Path): Output file path.
            
        Returns:
            bool: True if successful, False otherwise.
        """
        # List of Invidious instances to try
        instances = [
            "https://invidious.snopyta.org",
            "https://yewtu.be",
            "https://invidious.kavin.rocks",
            "https://vid.puffyan.us"
        ]
        
        for instance in instances:
            try:
                # Get video info from Invidious API
                api_url = f"{instance}/api/v1/videos/{video_id}"
                response = requests.get(api_url, timeout=10)
                
                if response.status_code != 200:
                    continue
                    
                data = response.json()
                
                # Find audio stream
                audio_streams = [f for f in data.get('adaptiveFormats', []) 
                               if f.get('type', '').startswith('audio')]
                
                if not audio_streams:
                    continue
                    
                # Get the best quality audio
                audio_stream = sorted(audio_streams, 
                                    key=lambda x: x.get('bitrate', 0), 
                                    reverse=True)[0]
                
                audio_url = audio_stream.get('url')
                
                if not audio_url:
                    continue
                    
                # Download the audio file
                audio_response = requests.get(audio_url, stream=True, timeout=30)
                
                if audio_response.status_code != 200:
                    continue
                    
                with open(output_path, 'wb') as f:
                    for chunk in audio_response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        
                return True
                
            except Exception as e:
                logger.warning(f"Invidious instance {instance} failed: {str(e)}")
                continue
                
        return False
    
    def _download_via_y2mate(self, video_id, output_path):
        """
        Download video using Y2mate service.
        
        Args:
            video_id (str): YouTube video ID.
            output_path (Path): Output file path.
            
        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            # This is a simplified implementation
            # In a real implementation, you would need to reverse engineer
            # the current Y2mate API which changes frequently
            
            # Step 1: Create a session and get a download link
            session = requests.Session()
            
            # Use a URL that's less likely to be blocked
            youtube_url = f"https://www.youtube.com/watch?v={video_id}"
            encoded_url = urllib.parse.quote(youtube_url)
            
            # Step 2: Get download links (this is a simplified example)
            # In reality, this would involve multiple requests and parsing responses
            api_url = f"https://www.y2mate.com/mates/analyze/ajax"
            
            payload = {
                'url': youtube_url,
                'q_auto': 0,
                'ajax': 1
            }
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
                'X-Requested-With': 'XMLHttpRequest',
                'Origin': 'https://www.y2mate.com',
                'Referer': f'https://www.y2mate.com/youtube/{video_id}'
            }
            
            # This is a simplified implementation
            # In reality, you would need to handle multiple requests and parse responses
            
            # For demonstration purposes, we'll use a more direct approach
            # that doesn't rely on the actual Y2mate API
            
            # Use a YouTube frontend that provides direct download links
            alternative_url = f"https://www.y2mate.com/youtube/{video_id}"
            
            # Since we can't fully implement the Y2mate API here,
            # we'll return False to try other methods
            return False
            
        except Exception as e:
            logger.warning(f"Y2mate download failed: {str(e)}")
            return False
    
    def _download_via_alternative_domain(self, video_id, output_path):
        """
        Download video using alternative YouTube domains.
        
        Args:
            video_id (str): YouTube video ID.
            output_path (Path): Output file path.
            
        Returns:
            bool: True if successful, False otherwise.
        """
        # Alternative domains that might bypass restrictions
        domains = [
            "youtube.com",
            "youtubepp.com",
            "youtube-nocookie.com"
        ]
        
        for domain in domains:
            try:
                url = f"https://www.{domain}/watch?v={video_id}"
                
                ydl_opts = {
                    'format': 'bestaudio/best',
                    'outtmpl': str(output_path),
                    'quiet': False,
                    'noplaylist': True,
                }
                
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([url])
                    
                if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                    return True
                    
            except Exception as e:
                logger.warning(f"Alternative domain {domain} failed: {str(e)}")
                continue
                
        return False
    
    def _download_direct_audio_url(self, video_id, output_path):
        """
        Attempt to download audio directly from known URL patterns.
        
        Args:
            video_id (str): YouTube video ID.
            output_path (Path): Output file path.
            
        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            # Try to get audio via a public API that doesn't require authentication
            api_url = f"https://pipedapi.kavin.rocks/streams/{video_id}"
            
            response = requests.get(api_url, timeout=10)
            
            if response.status_code != 200:
                return False
                
            data = response.json()
            
            # Find audio stream
            audio_streams = [s for s in data.get('audioStreams', []) 
                           if s.get('url')]
            
            if not audio_streams:
                return False
                
            # Get the best quality audio
            audio_stream = sorted(audio_streams, 
                                key=lambda x: int(x.get('bitrate', 0)), 
                                reverse=True)[0]
            
            audio_url = audio_stream.get('url')
            
            if not audio_url:
                return False
                
            # Download the audio file
            audio_response = requests.get(audio_url, stream=True, timeout=30)
            
            if audio_response.status_code != 200:
                return False
                
            with open(output_path, 'wb') as f:
                for chunk in audio_response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    
            return True
            
        except Exception as e:
            logger.warning(f"Direct audio URL download failed: {str(e)}")
            return False
    
    def _extract_audio(self, video_path):
        """
        Extract audio from video file.
        
        Args:
            video_path (str): Path to the video file.
            
        Returns:
            str: Path to the extracted audio file.
        """
        logger.info(f"Extracting audio from video: {video_path}")
        
        # Define output audio path
        audio_path = str(self.output_dir / "audio.wav")
        
        try:
            # Extract audio using ffmpeg
            (
                ffmpeg
                .input(video_path)
                .output(audio_path, acodec='pcm_s16le', ac=1, ar='16000')
                .run(quiet=True, overwrite_output=True)
            )
            
            logger.info(f"Audio extracted successfully to: {audio_path}")
            return audio_path
            
        except ffmpeg.Error as e:
            logger.error(f"FFmpeg error: {e.stderr.decode() if e.stderr else str(e)}")
            
            # Try alternative approach with simpler options
            try:
                logger.info("Trying alternative audio extraction approach...")
                os.system(f'ffmpeg -i "{video_path}" -vn -acodec pcm_s16le -ar 16000 -ac 1 "{audio_path}" -y')
                
                if os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
                    logger.info(f"Audio extracted successfully using alternative approach to: {audio_path}")
                    return audio_path
                else:
                    raise RuntimeError("Alternative audio extraction failed")
            except Exception as alt_e:
                logger.error(f"Alternative extraction error: {str(alt_e)}")
                raise RuntimeError(f"Failed to extract audio: {str(e)}")


# Example usage
if __name__ == "__main__":
    # Test with a sample video URL
    processor = VideoProcessor()
    try:
        audio_file = processor.extract_audio_from_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        print(f"Audio extracted to: {audio_file}")
    except Exception as e:
        print(f"Error: {str(e)}")