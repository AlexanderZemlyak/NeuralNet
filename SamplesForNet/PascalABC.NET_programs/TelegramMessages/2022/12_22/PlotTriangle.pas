uses PlotWPF;

var xx := Arr(2.0,3.0,1.0,2.0);
var yy := Arr(3.0,2.5,2.0,3.0);

begin
  LineGraphWPF.Create(xx,yy).AddMarkerGraph(xx,yy);
end.