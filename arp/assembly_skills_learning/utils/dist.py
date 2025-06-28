import socket
from contextlib import closing

def find_free_port():
    """Binds to all available network interfaces and automatically selects an available port:

    Returns:
        int: the free port number 
    """    
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        # Binds to all available network interfaces and Automatically selects an available port:
        # '' (empty string) or '0.0.0.0': 
        # This means the socket should bind to all available network interfaces on the machine. 
        # Essentially, it can accept connections from any IP address on the machine.
        # 0: This special value tells the operating system to automatically select an available port. 
        # The system will pick a free port from the range of available ports, 
        # and you can retrieve that chosen port by calling s.getsockname()[1].
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    
def find_free_port_for_tensorboard(start_port:int=6006):
    """
    Finds a free port in the specified range, starting from `start_port` to `start_port + 20`.
    If the port is unavailable, it tries the next one in the range.
    
    Args:
        start_port (int): The starting port number (default is tensorboard default port 6006).
    
    Returns:
        int: A free port number within the range of [start_port, start_port+20].
    """
    end_port = start_port + 20
    for port in range(start_port, end_port + 1):
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            try:
                s.bind(('localhost', port))  # Attempt to bind to the port
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                return s.getsockname()[1]  # If successful, return the port number
            except OSError:
                continue  # If the port is in use, try the next one
    print(f'Ports from range [{start_port, end_port}] are all unavailable.')

