"""Bound complete API calls even when a busy provider sends keepalive bytes."""
import time
import httpx


class DeadlineStream(httpx.SyncByteStream):
    def __init__(self,stream,deadline):self.stream,self.deadline=stream,deadline

    def __iter__(self):
        for chunk in self.stream:
            if time.monotonic()>self.deadline:
                self.stream.close()
                raise httpx.ReadTimeout('Absolute model request deadline exceeded')
            yield chunk

    def close(self):self.stream.close()


def bounded_http_client(seconds=75):
    def request_hook(request):request.extensions['model_deadline']=time.monotonic()+seconds
    def response_hook(response):
        response.stream=DeadlineStream(response.stream,response.request.extensions['model_deadline'])
    return httpx.Client(timeout=httpx.Timeout(connect=10,read=15,write=15,pool=10),
                        event_hooks={'request':[request_hook],'response':[response_hook]})
