using System;
using System.Collections.Generic;

namespace ITI.Train
{
    internal class Train : ITrain
    {
        private readonly string _name;
        private readonly ICompany _company;
        private ILine _assignment;

        internal Train( string name, ICompany company )
        {
            if( String.IsNullOrEmpty( name ) ) throw new ArgumentNullException();

            _name = name;
            _company = company;
        }

        public string Name { get => _name; }
        public ICompany Company { get => _company; }

        public ILine Assignment { get => _assignment; }

        public void AssignTo( ILine line )
        {
            if( line != null && line.City != _company.City ) throw new ArgumentException( "The line is not in the same city than the train." );

            if( _assignment != null ) ((List<Train>)_assignment.Trains).Remove( this );
            _assignment = line;
            if( line != null ) ((List<Train>)line.Trains).Add( this );
        }
    }
}
