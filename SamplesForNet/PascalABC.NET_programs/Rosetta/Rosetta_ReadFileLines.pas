// https://rosettacode.org/wiki/Read_a_file_line_by_line#PascalABC.NET

begin
  var f := OpenRead('_a.pas');
  while not f.Eof do
    Println(f.ReadString);
  f.Close
end.
