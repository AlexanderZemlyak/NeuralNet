##
var s := 'Console.WriteLine()';
var m := Regex.Match(s, 'Write(?:Line)?');   // WriteLine
Print(m.Groups[1]);
