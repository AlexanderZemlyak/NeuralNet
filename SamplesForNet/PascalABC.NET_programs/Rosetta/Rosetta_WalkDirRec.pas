// https://rosettacode.org/wiki/Walk_a_directory/Recursively#PascalABC.NET

begin
  var path := 'C:\PABCWork.NET';
  EnumerateAllFiles(path,'*.pas').PrintLines;
end.