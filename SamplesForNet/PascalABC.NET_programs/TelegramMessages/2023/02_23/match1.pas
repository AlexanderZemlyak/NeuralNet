##
var s := ' red    green  gray blue   yiellow  orange magenta ';
s.ToWords.Println;
foreach var m in s.Matches('\w+') do
  Println(m.Value,m.Index);
