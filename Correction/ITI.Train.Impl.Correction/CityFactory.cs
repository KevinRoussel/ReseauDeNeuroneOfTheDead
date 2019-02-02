namespace ITI.Train
{
    public static class CityFactory
    {
        public static ICity CreateCity( string name )
        {
            return new City( name );
        }
    }
}
