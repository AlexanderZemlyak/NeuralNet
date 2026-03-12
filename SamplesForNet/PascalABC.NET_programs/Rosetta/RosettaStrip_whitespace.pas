// https://rosettacode.org/wiki/Strip_control_codes_and_extended_characters_from_a_string#PascalABC.NET

begin
  var s := #9'  abc  '#9;
  Writeln(s.TrimStart,'|');
  Writeln(s.TrimEnd,'|');
  Writeln(s.Trim,'|');
end.