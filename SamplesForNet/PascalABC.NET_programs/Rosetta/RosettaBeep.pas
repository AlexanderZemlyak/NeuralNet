// https://rosettacode.org/wiki/Terminal_control/Ringing_the_terminal_bell#PascalABC.NET

uses CRT;

begin
  GotoXY(60,15);
  Print('PascalABC.NET');
  HideCursor;
  Sleep(1000);
  ShowCursor;
  Sleep(1000);
end.