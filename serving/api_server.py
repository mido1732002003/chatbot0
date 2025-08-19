"""HTTP API server for model serving."""

import json
import torch
from http.server import HTTPServer, BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs
from typing import Optional, Dict, Any
import threading
import signal
import sys
from pathlib import Path

from core.model import TransformerLM
from core.generation import generate
from alignment.safety_filter import SafetyFilter
from utils.logging_utils import setup_logger


class APIHandler(BaseHTTPRequestHandler):
    """HTTP request handler for API endpoints."""
    
    def do_GET(self):
        """Handle GET requests."""
        parsed_path = urlparse(self.path)
        
        if parsed_path.path == '/health':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            response = {
                'status': 'healthy',
                'model_loaded': self.server.model is not None,
                'device': str(self.server.device)
            }
            self.wfile.write(json.dumps(response).encode())
        else:
            self.send_error(404, 'Not Found')
            
    def do_POST(self):
        """Handle POST requests."""
        parsed_path = urlparse(self.path)
        
        if parsed_path.path == '/v1/generate':
            self.handle_generate()
        else:
            self.send_error(404, 'Not Found')
            
    def handle_generate(self):
        """Handle text generation request."""
        try:
            # Read request body
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            request_data = json.loads(post_data.decode('utf-8'))
            
            # Extract parameters
            prompt = request_data.get('prompt', '')
            max_tokens = request_data.get('max_tokens', 100)
            temperature = request_data.get('temperature', 0.8)
            top_k = request_data.get('top_k', 50)
            top_p = request_data.get('top_p', 0.9)
            repetition_penalty = request_data.get('repetition_penalty', 1.0)
            seed = request_data.get('seed', None)
            
            # Validate prompt
            if not prompt:
                self.send_error(400, 'Missing prompt')
                return
                
            # Check safety if enabled
            if self.server.safety_filter:
                prompt_check = self.server.safety_filter.check_prompt_safety(prompt)
                if not prompt_check['is_safe']:
                    # Return safe response
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    response = {
                        'generated_text': prompt_check['suggested_response'],
                        'token_count': 0,
                        'safety_filtered': True
                    }
                    self.wfile.write(json.dumps(response).encode())
                    return
                    
            # Tokenize prompt
            input_ids = torch.tensor(
                [self.server.tokenizer.bos_token_id] + self.server.tokenizer.encode(prompt),
                dtype=torch.long
            ).unsqueeze(0).to(self.server.device)
            
            # Generate
            with torch.no_grad():
                output = generate(
                    self.server.model,
                    input_ids,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    eos_token_id=self.server.tokenizer.eos_token_id,
                    pad_token_id=self.server.tokenizer.pad_token_id,
                    seed=seed
                )
                
            # Decode response
            generated_ids = output['generated_ids'][0]
            response_ids = generated_ids[input_ids.shape[1]:]
            generated_text = self.server.tokenizer.decode(response_ids.tolist())
            
            # Check response safety if enabled
            if self.server.safety_filter:
                response_check = self.server.safety_filter.check_response_safety(generated_text)
                if not response_check['is_safe']:
                    generated_text = response_check.get('filtered_response', "[Response filtered]")
                    
            # Send response
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            response = {
                'generated_text': generated_text,
                'token_count': output['generated_tokens'],
                'safety_filtered': False
            }
            self.wfile.write(json.dumps(response).encode())
            
        except json.JSONDecodeError:
            self.send_error(400, 'Invalid JSON')
        except Exception as e:
            self.server.logger.error(f"Generation error: {str(e)}", exc_info=True)
            self.send_error(500, f'Internal Server Error: {str(e)}')
            
    def log_message(self, format, *args):
        """Custom logging."""
        self.server.logger.info(f"{self.address_string()} - {format % args}")


class APIServer:
    """HTTP API server for model serving."""
    
    def __init__(
        self,
        model_path: str,
        tokenizer,
        host: str = '0.0.0.0',
        port: int = 8080,
        enable_safety: bool = True,
        device: Optional[torch.device] = None,
        logger: Optional[Any] = None
    ):
        """Initialize API server.
        
        Args:
            model_path: Path to model checkpoint
            tokenizer: Tokenizer instance
            host: Host to bind to
            port: Port to listen on
            enable_safety: Whether to enable safety filtering
            device: Device to run on
            logger: Logger instance
        """
        # Setup device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        # Load model
        self.model = TransformerLM.from_checkpoint(model_path, device=self.device)
        self.model.eval()
        self.tokenizer = tokenizer
        
        # Setup safety filter
        if enable_safety:
            self.safety_filter = SafetyFilter(enable_filter=True)
        else:
            self.safety_filter = None
            
        # Server configuration
        self.host = host
        self.port = port
        
        # Setup logger
        if logger is None:
            self.logger = setup_logger('api_server')
        else:
            self.logger = logger
            
        # Create HTTP server
        self.httpd = ThreadingHTTPServer((host, port), APIHandler)
        self.httpd.model = self.model
        self.httpd.tokenizer = self.tokenizer
        self.httpd.device = self.device
        self.httpd.safety_filter = self.safety_filter
        self.httpd.logger = self.logger
        
        # Setup graceful shutdown
        signal.signal(signal.SIGINT, self.shutdown_handler)
        signal.signal(signal.SIGTERM, self.shutdown_handler)
        
        self.logger.info(f"API Server initialized on {host}:{port}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Safety filter: {'enabled' if enable_safety else 'disabled'}")
        
    def start(self):
        """Start the API server."""
        self.logger.info(f"Starting API server on http://{self.host}:{self.port}")
        self.logger.info("Endpoints:")
        self.logger.info("  GET  /health - Health check")
        self.logger.info("  POST /v1/generate - Generate text")
        
        try:
            self.httpd.serve_forever()
        except KeyboardInterrupt:
            self.shutdown()
            
    def shutdown(self):
        """Shutdown the server gracefully."""
        self.logger.info("Shutting down API server...")
        self.httpd.shutdown()
        self.httpd.server_close()
        self.logger.info("API server stopped")
        
    def shutdown_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.shutdown()
        sys.exit(0)