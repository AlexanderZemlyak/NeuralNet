// https://rosettacode.org/wiki/Stack_traces#PascalABC.NET

procedure qqq;
begin
  var st := new System.Diagnostics.StackTrace();
  Print(st);
end;

procedure ppp;
begin
  qqq;
end;

begin
  ppp
end.