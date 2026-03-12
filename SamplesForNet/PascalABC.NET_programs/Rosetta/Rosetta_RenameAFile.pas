// https://rosettacode.org/wiki/Rename_a_file#PascalABC.NET

uses System.IO;

begin
  &File.Move('input.txt','output.txt');
  &File.Move('\input.txt','\output.txt');

  Directory.Move('docs','mydocs');
  Directory.Move('\docs','\mydocs');
end.