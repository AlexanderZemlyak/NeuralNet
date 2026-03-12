##
function SieveOfEratosthenes(n: integer): array of integer;
begin
  var isPrime := [False] * 2 + [True] * (n - 1); 

  for var i := 2 to ISqrt(n) do
    if isPrime[i] then
      for var j := i * i to n step i do
        isPrime[j] := False;

  Result := (2..n).Where(i -> isPrime[i]).ToArray;
end;

SieveOfEratosthenes(1000).Println;
