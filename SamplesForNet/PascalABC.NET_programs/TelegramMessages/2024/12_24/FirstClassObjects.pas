begin
  var op := new Dictionary<string,(real,real) -> real>;
  op['add'] := (x,y) -> x + y;
  op['mult'] := (x,y) -> x * y;

  op['add'](2,3).Print;
  op['mult'](2,3).Print;
end.