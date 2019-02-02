namespace ITI.Train
{
    internal class DoubleLinkedNode<T>
    {
        internal DoubleLinkedNode( T value )
        {
            Value = value;
        }
        internal T Value { get; private set; }
        internal DoubleLinkedNode<T> Next { get; set; }
        internal DoubleLinkedNode<T> Previous { get; set; }
        internal void SetPrevious( DoubleLinkedNode<T> previous )
        {
            Previous = previous;
        }
        internal void SetNext( DoubleLinkedNode<T> next )
        {
            Next = next;
        }
    }
}
