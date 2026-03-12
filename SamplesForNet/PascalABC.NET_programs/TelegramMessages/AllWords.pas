##
var letters := 'ооаамсптр';
var ld := letters.EachCount;
var AllWords := ReadAllLines('C:\PABCWork.NET\Samples\Games\BookWorm\words.txt');
foreach var word in AllWords do
begin
  if word.Length<4 then
    continue;
  var wd := word.EachCount;
  if wd.Keys.All(c -> wd[c]<=ld.Get(c)) then
    Println(word);
end;