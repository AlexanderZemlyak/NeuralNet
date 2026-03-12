// https://rosettacode.org/wiki/Determine_if_a_string_is_numeric

function IsNumeric(s: string): boolean;
begin
  var i: integer;
  Result := integer.TryParse(s,i)
end;

begin
  var s := '123';
  if IsInteger(s) then
    Print('string is numeric')
end.