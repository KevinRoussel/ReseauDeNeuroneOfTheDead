using System;
using System.Collections.Generic;
using System.Linq;

namespace ITI.Train
{
    internal class City : ICity
    {
        private readonly List<Company> _companies;
        private readonly List<Line> _lines;
        private readonly List<Station> _stations;

        internal City( string name )
        {
            if( String.IsNullOrEmpty( name ) ) throw new ArgumentNullException();

            _companies = new List<Company>();
            _lines = new List<Line>();
            _stations = new List<Station>();
            Name = name;
        }

        public string Name { get; private set; }

        public ICompany AddCompany( string name )
        {
            if( String.IsNullOrEmpty( name ) ) throw new ArgumentNullException();

            if( _companies.Where( c => c.Name == name ).Any() )
            {
                throw new ArgumentException( "There is already a company with this name." );
            }
            Company company = new Company( name, this );
            _companies.Add( company );
            return company;
        }

        public ILine AddLine( string name )
        {
            if( String.IsNullOrEmpty( name ) ) throw new ArgumentNullException();

            if( _lines.Where( l => l.Name == name ).Any() )
            {
                throw new ArgumentException( "There is already a line with this name." );
            }
            Line line = new Line( name, this );
            _lines.Add( line );
            return line;
        }

        public IStation AddStation( string name, int x, int y )
        {
            if( _stations.Where( s => s.Name == name ).Any() || _stations.Where( s => s.X == x && s.Y == y ).Any() )
            {
                throw new ArgumentException( "There is already a station with this name or there is already a station at theses coordonates" );
            }
            Station station = new Station( name, x, y, this );
            _stations.Add( station );
            return station;
        }

        public ICompany FindCompany( string name )
        {
            return _companies.Where( c => c.Name == name ).FirstOrDefault();
        }

        public ILine FindLine( string name )
        {
            return _lines.Where( c => c.Name == name ).FirstOrDefault();

        }

        public IStation FindNearestStation( int x, int y )
        {
            if( _stations.Count == 0 )
                return null;

            double xDist;
            double yDist;
            List<KeyValuePair<Station, double>> stationCoord = new List<KeyValuePair<Station, double>>();

            foreach( Station station in _stations )
            {
                if( station.X == x && station.Y == y )
                    return station;

                xDist = Math.Pow( Math.Max( x, station.X ) - Math.Min( x, station.X ), 2 );
                yDist = Math.Pow( Math.Max( y, station.Y ) - Math.Min( y, station.Y ), 2 );

                stationCoord.Add( new KeyValuePair<Station, double>( station, Math.Sqrt( xDist + yDist ) ) );
            }

            stationCoord.Sort( ( xx, yy ) => xx.Value.CompareTo( yy.Value ) );

            if( stationCoord.Count > 1 && stationCoord[ 0 ].Value == stationCoord[ 1 ].Value )
            {
                return stationCoord.LastOrDefault().Key;
            }
            return stationCoord.FirstOrDefault().Key;
        }

        public IStation FindStation( string name )
        {
            return _stations.Where( c => c.Name == name ).FirstOrDefault();
        }

        public IReadOnlyList<IStation> GetFastestPath( IStation from, IStation to )
        {
            if( from == null || to == null ) throw new ArgumentException();
            var graph = new Graph();
            foreach( var s in _stations )
            {
                graph.AddNode( s.Name );
            }

            foreach( var l in _lines )
            {
                foreach( var s in l.Stations.Where( x => l.Next( x ) != null ) )
                {
                    var next = l.Next( s );
                    var dist = Math.Sqrt( Math.Pow( (Math.Max( s.X, next.X ) - Math.Min( s.X, next.X )), 2 ) + Math.Pow( (Math.Max( s.Y, next.Y ) - Math.Min( s.Y, next.Y )), 2 ) );
                    graph.AddConnection( s.Name, next.Name, dist, true );
                }
            }
            var res = graph.Do( from.Name, to.Name );
            List<IStation> ret = new List<IStation>();

            if( double.IsInfinity( res.Item1 ) ) return ret;

            foreach( var stationName in res.Item2.Split( new char[] { '$' }, StringSplitOptions.RemoveEmptyEntries ) )
            {
                ret.Add( _stations.Find( x => x.Name == stationName ) );
            }
            return ret;
        }
        public IReadOnlyList<IStation> GetShortestPath( IStation from, IStation to )
        {
            if( from == null || to == null ) throw new ArgumentException();
            var graph = new Graph();
            foreach( var s in _stations )
            {
                graph.AddNode( s.Name );
            }

            foreach( var l in _lines )
            {
                foreach( var s in l.Stations.Where( x => l.Next( x ) != null ) )
                {
                    var next = l.Next( s );
                    var dist = 1;
                    graph.AddConnection( s.Name, next.Name, dist, true );
                }
            }

            var res = graph.Do( from.Name, to.Name );
            List<IStation> ret = new List<IStation>();

            if( double.IsInfinity( res.Item1 ) ) return ret;

            foreach( var stationName in res.Item2.Split( new char[] { '$' }, StringSplitOptions.RemoveEmptyEntries ) )
            {
                ret.Add( _stations.Find( x => x.Name == stationName ) );
            }
            return ret;
        }
        public double TravelTimeOnSameLine( IStation from, IStation to, int speed )
        {
            if( from == to || from == null || to == null || speed < 1 ) throw new ArgumentException();

            Line line = _lines.Where( u => u.Stations.Contains( from ) && u.Stations.Contains( to ) ).FirstOrDefault();

            if( line == null ) throw new ArgumentException();

            IStation current = from;
            double dist = 0;
            while( line.Next( current ) != null )
            {
                IStation next = line.Next( current );
                double xDist = Math.Pow( Math.Max( current.X, next.X ) - Math.Min( current.X, next.X ), 2 );
                double yDist = Math.Pow( Math.Max( current.Y, next.Y ) - Math.Min( current.Y, next.Y ), 2 );
                dist += Math.Sqrt( xDist + yDist );
                if( next == to ) return dist / speed;
                current = next;
            }
            current = from;
            while( line.Previous( current ) != null )
            {
                IStation previous = line.Previous( current );
                double xDist = Math.Pow( Math.Max( current.X, previous.X ) - Math.Min( current.X, previous.X ), 2 );
                double yDist = Math.Pow( Math.Max( current.Y, previous.Y ) - Math.Min( current.Y, previous.Y ), 2 );
                dist += Math.Sqrt( xDist + yDist );
                if( previous == to ) return dist / speed;
                current = previous;
            }
            throw new InvalidOperationException();
        }
    }
}
