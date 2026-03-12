// https://rosettacode.org/wiki/Strip_control_codes_and_extended_characters_from_a_string#PascalABC.NET

{$zerobasedstrings}
begin
  var s := '0123456789';
  Writeln(s[1:]);
  Writeln(s[:^1]);
  Writeln(s[1:^1]);
end.