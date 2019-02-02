using System;
using System.Collections.Generic;

namespace ITI.Train
{
    internal class Node
    {
        IList<NodeConnection> _connections;

        internal string Name { get; private set; }

        internal double DistanceFromStart { get; set; }
        internal string PathFromStart { get; set; }
        internal IEnumerable<NodeConnection> Connections
        {
            get { return _connections; }
        }

        internal Node( string name )
        {
            Name = name;
            _connections = new List<NodeConnection>();
        }

        internal void AddConnection( Node targetNode, double distance, bool twoWay )
        {
            if( targetNode == null ) throw new ArgumentNullException( "targetNode" );
            if( targetNode == this )
                throw new ArgumentException( "Node may not connect to itself." );
            if( distance <= 0 ) throw new ArgumentException( "Distance must be positive." );

            _connections.Add( new NodeConnection( targetNode, distance ) );
            if( twoWay ) targetNode.AddConnection( this, distance, false );
        }
    }
    internal class NodeConnection
    {
        internal Node Target { get; private set; }
        internal double Distance { get; private set; }
        internal NodeConnection( Node target, double distance )
        {
            Target = target;
            Distance = distance;
        }
    }
}
