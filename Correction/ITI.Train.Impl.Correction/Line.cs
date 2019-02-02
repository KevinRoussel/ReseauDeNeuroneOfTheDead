using System;
using System.Collections.Generic;

namespace ITI.Train
{
    internal class Line : ILine
    {
        private readonly List<Train> _trains;
        private DoubleLinkedList<Station> _stations;

        internal Line( string name, ICity city )
        {
            if( String.IsNullOrEmpty( name ) ) throw new ArgumentException();

            _trains = new List<Train>();
            _stations = new DoubleLinkedList<Station>();
            Name = name;
            City = city;
        }
        public string Name { get; private set; }
        public ICity City { get; private set; }
        public IEnumerable<IStation> Stations => _stations.ToList();

        public IEnumerable<ITrain> Trains => _trains;

        public void AddBefore( IStation toAdd, IStation before = null )
        {

            if( toAdd == null ) throw new ArgumentException();
            if( _stations.Contains( (Station)toAdd ) ) throw new ArgumentException();

            if( before == null ) _stations.AddBefore( (Station)toAdd );
            else
            {
                DoubleLinkedNode<Station> node = _stations.Find( (Station)before );
                _stations.AddBefore( (Station)toAdd, node );
            }
            ((Station)toAdd).AddLine( this );
        }

        public IStation Next( IStation s )
        {
            if( s == null ) throw new ArgumentException();
            if( !_stations.Contains( (Station)s ) ) throw new ArgumentException();
            return _stations.Find( (Station)s ).Next?.Value;
        }

        public IStation Previous( IStation s )
        {
            if( s == null ) throw new ArgumentException();
            if( !_stations.Contains( (Station)s ) ) throw new ArgumentException();

            return _stations.Find( (Station)s ).Previous?.Value;
        }

        public void Remove( IStation s )
        {
            if( s == null ) throw new ArgumentException();
            if( !_stations.Contains( (Station)s ) ) throw new ArgumentException();
            _stations.Remove( (Station)s );
            ((Station)s).RemoveLine( this );
        }
    }
}
