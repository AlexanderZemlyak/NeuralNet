##
uses Utils;
var n := 1000000;
var Sq := ArrRandomInteger(n,0,MaxInt-1);

Benchmark(()->begin
  var sq1 := sq.Order.ToArray;
end).Println;  

Benchmark(()->begin
  Sort(sq)
end).Println;  
  