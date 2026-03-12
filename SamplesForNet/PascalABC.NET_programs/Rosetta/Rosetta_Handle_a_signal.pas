// https://rosettacode.org/wiki/Handle_a_signal#PascalABC.NET

begin
  var counter := 0;
  while True do
  begin
    Print(counter);
    counter += 1;
    Sleep(500);
  end;
  Console.CancelKeyPress += (o,e) -> begin
    Println(Milliseconds);
    halt
  end;
end.