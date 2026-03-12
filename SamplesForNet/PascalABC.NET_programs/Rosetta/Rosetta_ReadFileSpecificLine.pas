// https://rosettacode.org/wiki/Read_a_specific_line_from_a_file#PascalABC.NET

begin
  var linenum := 3;
  ReadLines('_a.pas').Skip(linenum-1).First.Print;
end.
