// https://rosettacode.org/wiki/Read_a_file_character_by_character/UTF8#PascalABC.NET

begin
  var f := OpenRead('a.txt',Encoding.UTF8);
  while not f.Eof do
    Print(f.ReadChar);
  f.Close
end.
