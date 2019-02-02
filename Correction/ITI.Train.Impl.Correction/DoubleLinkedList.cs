using System;
using System.Collections.Generic;

namespace ITI.Train
{
    internal class DoubleLinkedList<T>
    {
        DoubleLinkedNode<T> _head;
        DoubleLinkedNode<T> _tail;

        internal void AddBefore( T toAdd, DoubleLinkedNode<T> before = null )
        {
            if( toAdd == null ) throw new ArgumentException();
            if( Contains( toAdd ) ) throw new ArgumentException();
            var newNode = new DoubleLinkedNode<T>( toAdd );
            if( before != null && !Contains( before ) ) throw new ArgumentException();

            if( before == null )
            {
                if( _head == null )
                {
                    _head = newNode;
                    _tail = newNode;
                }
                else
                {
                    _head.SetNext( newNode );
                    newNode.SetPrevious( _head );
                    _head = newNode;
                }
            }
            else
            {
                if( before == _tail )
                {
                    before.SetPrevious( newNode );
                    newNode.SetNext( before );
                    _tail = newNode;
                }
                else
                {
                    before.Previous.SetNext( newNode );
                    newNode.SetPrevious( before.Previous );
                    newNode.SetNext( before );
                    before.SetPrevious( newNode );
                }
            }
        }
        internal void Remove( T value )
        {
            if( value == null ) throw new ArgumentException();
            if( !Contains( value ) ) throw new ArgumentException();
            var node = Find( value );
            if( node.Previous != null ) node.Previous.SetNext( node.Next );
            else _tail = node.Next;
            if( node.Next != null ) node.Next.SetPrevious( node.Previous );
            else _head = node.Previous;

        }
        internal bool Contains( T value )
        {
            DoubleLinkedNode<T> next = _tail;
            while( next != null )
            {
                if( next.Value.Equals( value ) ) return true;
                next = next.Next;
            }
            return false;
        }
        internal bool Contains( DoubleLinkedNode<T> node )
        {
            DoubleLinkedNode<T> next = _tail;
            while( next != null )
            {
                if( next == node ) return true;
                next = next.Next;
            }
            return false;
        }
        internal DoubleLinkedNode<T> Find( T value )
        {
            DoubleLinkedNode<T> next = _tail;
            while( next != null )
            {
                if( next.Value.Equals( value ) ) return next;
                next = next.Next;
            }
            return null;
        }
        internal List<T> ToList()
        {
            DoubleLinkedNode<T> next = _tail;
            List<T> list = new List<T>();
            while( next != null )
            {
                list.Add( next.Value );
                next = next.Next;
            }
            return list;
        }
    }
}
