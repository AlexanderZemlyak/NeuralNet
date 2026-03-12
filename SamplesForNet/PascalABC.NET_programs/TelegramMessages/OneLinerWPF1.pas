##
uses GraphWPF;
(0..800).Cartesian(0..600).ForEach(\(x,y)->begin SetPixel(x,y,RGB(x+y,2*x-y,y+3*x)) end);