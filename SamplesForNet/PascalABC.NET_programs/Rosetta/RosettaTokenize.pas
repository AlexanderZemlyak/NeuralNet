// https://rosettacode.org/wiki/Tokenize_a_string#PascalABC.NET

begin
  var s := 'Hello,How,Are,You,Today';
  var strings := s.Split(',');
  Print(strings.JoinToString('.'));
end.
