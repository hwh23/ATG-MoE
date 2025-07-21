import threading
import queue
import time
from . import message 
from .logging_config import get_logger, log_prefix
logger = get_logger(__name__, False)


class Connection:
    def __init__(self, conn, max_queue:int, address:str, proxy:str="json", queue_time_out:float=0.01):
        self.conn = conn
        self.send_queue = queue.Queue(maxsize=max_queue)
        self.recv_queue = queue.Queue(maxsize=max_queue)
        self.proc_queue = queue.Queue(maxsize=max_queue)
        self.queue_time_out = queue_time_out
        self.addr = address
        self.print_address = f"{address[0]}:{address[1]}"
        self.proxy = proxy
        
        self.stop_event = threading.Event()
        self.recv_thread = threading.Thread(target=self._recv_loop, daemon=True)
        self.recv_thread.start()
    
    def __del__(self):
        self.close()
        del self.send_queue
        del self.recv_queue
        del self.proc_queue

    def close(self):
        self.stop_event.set()
        self.conn.close()
        self.send_queue = queue.Queue()
        self.recv_queue = queue.Queue()
        self.proc_queue = queue.Queue()
        self.recv_thread.join()

        
    def _recv_loop(self):
        while not self.stop_event.is_set():
            try:
                self.receive()
            except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
                logger.warning(f"{log_prefix('recv', self.addr)} | Disconnected")
                self.stop_event.set()
            except Exception as e:
                logger.error(f'{log_prefix("recv", self.addr)}  | Error: {e}')
                self.stop_event.set()
    
    def send(self)->None:
        try:
            obj = self.send_queue.get(timeout=self.queue_time_out)
            logger.info(f'{log_prefix("send", self.addr)} | Start  frame {obj["frame_idx"]}')
        except queue.Empty:
            return
        if self.proxy == 'json':
            message.send_json(self.conn, obj)
        else:
            message.send_pickle(self.conn, obj)
        logger.info(f'{log_prefix("send", self.addr)} | Finish frame {obj["frame_idx"]}')
    
    def receive(self)->dict:
        logger.info(f'{log_prefix("send", self.addr)} | Start')
        if self.proxy == 'json':
            data_dict = message.recv_json(self.conn)
        else:
            data_dict = message.recv_pickle(self.conn)
        self.recv_queue.put(data_dict)
        logger.info(f'{log_prefix("send", self.addr)} | Finish frame {data_dict["frame_idx"]}')
    
    
class ConnectionQueues:
    def __init__(self, queue_time_out:float=0.01):
        self.queue_time_out = queue_time_out
        self.connections:list[Connection] = []
        self.remove_thread = threading.Thread(target=self.remove, daemon=True)
        self.remove_thread.start()
    
    def __del__(self):
        self.connections.clear()
        self.remove_thread.join()
    
    def __len__(self):
        return len(self.connections)
    
    def __iter__(self):
        """Makes the class iterable (for conn in queue)"""
        return iter(self.connections)
        
    def get(self, idx:int=0, keep_item=True):
        if len(self.connections) > 0:
            if keep_item:
                return self.connections[idx]    
            else:
                return self.connections.pop(idx)
        else:
            logger.error('Cannot get connection when the queue is empty')
            raise ValueError
        
    def put(self, conn:Connection):
        conn.queue_time_out = self.queue_time_out
        self.connections.append(conn)
    
    def remove(self):
        while True:
            for idx, conn in enumerate(self.connections):
                if conn.stop_event.is_set():
                    logger.info(f'{log_prefix("Connection", conn.print_address)} | Removing connection')
                    conn.close()
                    self.connections.pop(idx)
            time.sleep(0.01)
