from queue import Queue
from threading import Event
import numpy as np
from anypost.presentation import RetainedResultView
from test_presentation import Reader,Viewer,drain


class BeamReader(Reader):
    def geometry(self):
        return dict(positions=np.array([[-1.,-.6,0.],[1.,-.6,0.],[-1.,.6,0.],[1.,.6,0.]]),triangles=np.empty((0,3),dtype=int),lines=np.array([[0,1],[2,3]]),line_to_element=np.array([1,0]),node_ids=np.array([7,3,9,12]),element_ids=np.array([901,305]))
    def components(self):return [dict(quantity_id="stress",component="x",location="element")]
    def read_field(self,*args,selection=None):
        values=np.array([10.,20.]);values.setflags(write=False);return values
    def associations(self,*args,selection=None):return np.array([305,901])


def test_line_adapter_prefetch_exact_ids_budget_and_retained_latest_close():
    reader,viewer,queue=BeamReader(),Viewer(),Queue();geometry=reader.geometry();descriptors=reader.components()
    for values in geometry.values():values.setflags(write=False)
    reader.geometry=reader.components=lambda:(_ for _ in ()).throw(AssertionError("owner performed IO"))
    presented=[];view=RetainedResultView(reader,viewer,queue.put,on_presented=presented.append,geometry=geometry,descriptors=descriptors)
    original=view.handle.mesh;assert original.element_count==2
    view.request_component("first","stress","x");drain(queue);viewer.present()
    np.testing.assert_array_equal(view.handle.mesh.element_scalars,[20.,10.])
    assert view.handle.mesh.lines is original.lines and view.handle.mesh.line_to_element is original.line_to_element
    assert view.handle.mesh.positions is original.positions and view.handle.generations.topology==0
    assert sum(reader.reservations.values())==8 and presented[-1].case_id=="first"
    entered,release=Event(),Event();original_read=reader.read_field
    def slow(case,*args,selection=None):
        if case=="obsolete":entered.set();assert release.wait(2)
        return original_read(case,*args,selection=selection)
    reader.read_field=slow
    view.request_component("obsolete","stress","x");assert entered.wait(2)
    view.request_component("latest","stress","x");release.set();drain(queue);viewer.present()
    assert view.last_presented.case_id=="latest"
    view.close();assert view.handle.removed and sum(reader.reservations.values())==0
