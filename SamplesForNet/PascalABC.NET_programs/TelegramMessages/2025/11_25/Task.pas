uses System.Threading.Tasks;

begin
  var t1 := Task.Run(()->
  begin
    var s: BigInteger := 0;
    for var i:=1 to 50_000_000 do
      s += i;
    Result := s;
  end);

  var t2 := Task.Run(()->
  begin
    var s: BigInteger := 0;
    for var i:=1 to 50_000_000 do
      s += i*i;
    Result := s;
  end);

  Println(t1.Result + t2.Result);
end.
