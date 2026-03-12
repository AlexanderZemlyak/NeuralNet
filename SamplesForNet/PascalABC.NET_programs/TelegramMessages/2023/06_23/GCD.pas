##
var (a,b) := (99945,6570);
while b>0 do
  (a,b) := (b,a mod b);
Print(a);