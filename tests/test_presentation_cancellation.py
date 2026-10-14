from queue import Queue
from threading import Event
from anypost import RetainedResultView
from test_presentation import Reader,Viewer,drain


def test_explicit_cancellation_capability_forwards_same_loader_event():
    class CancellableReader(Reader):
        supports_cancellation=True
        def __init__(self):super().__init__();self.events=[]
        def read_field(self,*args,selection=None,cancel_event=None):
            assert isinstance(cancel_event,Event);self.events.append(cancel_event)
            return super().read_field(*args,selection=selection)
        def associations(self,*args,selection=None,cancel_event=None):
            self.events.append(cancel_event)
            return super().associations(*args,selection=selection)
    reader,viewer,queue=CancellableReader(),Viewer(),Queue()
    view=RetainedResultView(reader,viewer,queue.put)
    view.request_component('case','displacement','ux');drain(queue)
    assert len(reader.events)==2 and reader.events[0] is reader.events[1] and view.error is None
    view.close()
