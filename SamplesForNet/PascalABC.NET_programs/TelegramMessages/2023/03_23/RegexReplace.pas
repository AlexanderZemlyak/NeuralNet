##
var s := 'один  два    четыре';
var s1 := Regex.Replace(s, ' +', ' ');
var s2 := Regex.Replace(s, '\w+', '<$0>');
var s3 := Regex.Replace(s, '\w+', m -> m.Value.ToUpper());
var s4 := Regex.Replace(s, '\w+', m -> m.Value + '(' + m.Length + ')');
|s1,s2,s3,s4|.PrintLines;