##
var s := 'ahaha';
foreach var m in s.Matches('ah*a') do
  Print(m.Index);
Println;
foreach var m in s.Matches('ah*+a') do
  Print(m.Index);

