// https://rosettacode.org/wiki/Concurrent_computing#PascalABC.NET

uses System.Threading.Tasks;

begin
  Task.Run(() -> Print('Enjoy'));  
  Task.Run(() -> Print('Rosetta'));  
  Task.Run(() -> Print('Code'));
  Sleep(100);
end.