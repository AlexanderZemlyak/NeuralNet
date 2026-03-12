// https://rosettacode.org/wiki/URL_encoding#PascalABC.NET

##
function URLEncode(s: string) := System.Uri.EscapeDataString(s);

Println(URLEncode('http://foo bar/'));
