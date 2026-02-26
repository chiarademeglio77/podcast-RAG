import time
import sys
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import src.config as config
from src.indexer import Indexer

class NewFileHandler(FileSystemEventHandler):
    def __init__(self, indexer: Indexer):
        self.indexer = indexer

    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith(".txt"):
            print(f"New file detected: {event.src_path}")
            # Optional: Add a small delay to ensure file is completely written
            time.sleep(1)
            self.indexer.index_files(directory=str(config.ARTICLES_DIR))

def start_monitor():
    indexer = Indexer()
    event_handler = NewFileHandler(indexer)
    observer = Observer()
    observer.schedule(event_handler, str(config.ARTICLES_DIR), recursive=False)
    
    print(f"Monitoring folder: {config.ARTICLES_DIR}")
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    start_monitor()
