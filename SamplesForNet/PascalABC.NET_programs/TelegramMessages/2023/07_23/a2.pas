uses Utils;

begin
  var n := 1000000000;
  Benchmark(procedure -> begin
    var mx := -integer.MaxValue;
    for var i:=1 to n do
      mx := Max(i,mx);
  end,1).Print;  

  Benchmark(procedure -> begin
    var mx := -integer.MaxValue;
    for var i:=1 to n do
      if i > mx then
        mx := i;
  end,1).Print;  
end.