uses School;
uses System.Threading.Tasks;

begin
  var primes := new List<integer>;
  var n := 10000000;
  MillisecondsDelta;
  for var i:=1 to n do
    if i.IsPrime then
        primes.Add(i);
  Println(primes.Count,MillisecondsDelta);
  primes.Clear;
  MillisecondsDelta;
  Parallel.For(1,n,i -> begin
    if i.IsPrime then
      lock primes do // гарантирует, что следующий оператор будет выполняться только одним потоком
        primes.Add(i);
  end);
  Println(primes.Count,MillisecondsDelta);
end.