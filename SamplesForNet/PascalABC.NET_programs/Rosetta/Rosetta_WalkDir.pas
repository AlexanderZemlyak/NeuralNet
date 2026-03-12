// https://rosettacode.org/wiki/Walk_a_directory/Non-recursively#PascalABC.NET

begin
  var path := 'C:\PABCWork.NET';
  EnumerateFiles(path,'*.pas').PrintLines;
end.