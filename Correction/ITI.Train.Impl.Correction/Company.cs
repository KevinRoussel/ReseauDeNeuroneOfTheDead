using System;
using System.Collections.Generic;
using System.Linq;

namespace ITI.Train
{
    internal class Company : ICompany
    {
        private readonly List<Train> _trains;
        internal Company( string name, ICity city )
        {
            if( String.IsNullOrEmpty( name ) ) throw new ArgumentException();

            _trains = new List<Train>();
            Name = name;
            City = city;

        }
        public string Name { get; private set; }

        public ICity City { get; private set; }

        public ITrain AddTrain( string name )
        {
            if( _trains.Where( t => t.Name == name ).Any() ) throw new ArgumentException( "There is already a train with this name." );

            Train train = new Train( name, this );
            _trains.Add( train );
            return train;
        }

        public ITrain FindTrain( string name )
        {
            return _trains.Where( c => c.Name == name ).FirstOrDefault();
        }
    }
}
