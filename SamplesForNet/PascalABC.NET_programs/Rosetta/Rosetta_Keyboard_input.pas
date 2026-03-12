// https://rosettacode.org/wiki/Keyboard_input/Obtain_a_Y_or_N_response#PascalABC.NET

begin
  repeat
    var q := console.ReadKey;
    if q.Key in |System.ConsoleKey.Y,System.ConsoleKey.N| then
    begin  
      Print($'{q.Key} pressed');
      exit
    end;  
  until False;
end.

