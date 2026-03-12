// https://rosettacode.org/wiki/File_size
uses System.IO;

begin
  FileInfo.Create('input.txt').Length.Println;
  FileInfo.Create('/input.txt').Length.Println;
end.