##
procedure PrintMatches(s,pattern: string) := 
  s.Matches(pattern).Select(m->(m.Index,m.Value)).Println;

var s := 'бойкот котомка кот';
PrintMatches(s,'\bкот\b'); 
PrintMatches(s,'кот\b');    
PrintMatches(s,'\bкот'); 