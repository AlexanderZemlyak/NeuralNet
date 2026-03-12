// https://rosettacode.org/wiki/Find_common_directory_path#PascalABC.NET

function CommonPrefix(arrs: array of array of string): array of string;
begin
  var min: integer := arrs.Min(a -> a.Length);
  // Транспонируем эту часть
  var at := ArrGen(min, j -> ArrGen(arrs.Length, i -> arrs[i][j]));
  Result := at.TakeWhile(x -> x.Skip(1).All(y -> y = x.First)).Select(a -> a[0]).ToArray
end;

function CommonDirectoryPath(paths: array of string; sep: char := '/'): string;
begin
  var arrs := paths.Select(a -> a.Split(sep)).ToArray;
  Result := CommonPrefix(arrs).JoinToString(sep);
end;

begin
  Print(CommonDirectoryPath(|
   '/home/user1/tmp/coverage/test',
   '/home/user1/tmp/covert/operator',
   '/home/user1/tmp/coven/members'|))
end.