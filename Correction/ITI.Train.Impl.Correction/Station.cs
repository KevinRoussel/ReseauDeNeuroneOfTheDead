using System;
using System.Collections.Generic;

namespace ITI.Train
{
    internal class Station : IStation
    {
        private readonly List<Line> _lines;

        internal Station( string name, int x, int y, ICity city )
        {
            if( String.IsNullOrEmpty( name ) ) throw new ArgumentException();
            _lines = new List<Line>();
            Name = name;
            X = x;
            Y = y;
            City = city;
        }

        public string Name { get; private set; }
        public ICity City { get; private set; }
        public int X { get; private set; }
        public int Y { get; private set; }
        public IEnumerable<ILine> Lines => _lines;
        internal void AddLine( Line l )
        {
            _lines.Add( l );
        }
        internal void RemoveLine( Line l )
        {
            _lines.Remove( l );
        }
    }
}
