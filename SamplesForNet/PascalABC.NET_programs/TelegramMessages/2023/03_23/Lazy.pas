##
var s := '<a href="helloworld.htm" title="Привет, Мир">Привет, Мир</a>';
s.Matches('".*"').PrintLines;
Println('-'*40);
s.Matches('".*?"').PrintLines;
